{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections       #-}
{-# LANGUAGE TypeFamilies        #-}
{-# LANGUAGE TypeOperators       #-}

import qualified Control.Foldl                as F
import           Control.Lens
import           Control.Monad.Random         (getRandomR, getRandomRs)
import qualified Data.Attoparsec.Text         as A
import           Data.Semigroup               ((<>))
import           Data.Singletons.Prelude.List (Head, Last)
import           Grenade
import           Grenade.Recurrent
import qualified Numeric.LinearAlgebra.Static as SA
import           Options.Applicative
import           Pipes
import           Pipes.Group
import qualified Pipes.Prelude                as P
import qualified Pipes.Text                   as PT
import qualified Pipes.Text.IO                as PT

-- The defininition for our simple recurrent network.
-- This file just trains a network to generate a repeating sequence
-- of 0 0 1.
--
-- The F and R types are Tagging types to ensure that the runner and
-- creation function know how to treat the layers.
type R = Recurrent
type F = FeedForward
type TRNNIn = 4
type TRNNOut = 10
type TFFNOut = 1
type TRNN = R (LSTM TRNNIn TRNNOut)
type TFFN = F (FullyConnected TRNNOut TFFNOut)
-- type NJetInputs = 4
-- type NJetOutputs = 14
-- type JetLayer = Concat ('D1 4) Trivial ('D1 4) TracksLayer

type TracksNetShape = '[ TRNN, TFFN ]

type TracksNet =
  RecurrentNetwork
    TracksNetShape
    '[ 'D1 TRNNIn, 'D1 TRNNOut, 'D1 TFFNOut ]

type TracksInput = RecurrentInputs TracksNetShape


data FeedForwardOpts = FeedForwardOpts Int LearningParameters

feedForward' :: Parser FeedForwardOpts
feedForward' =
  FeedForwardOpts
    <$> option auto (long "examples" <> short 'e' <> value 40000)
    <*> ( LearningParameters
          <$> option auto (long "train_rate" <> short 'r' <> value 0.01)
          <*> option auto (long "momentum" <> value 0.9)
          <*> option auto (long "l2" <> value 0.0005)
        )


parseJet :: A.Parser [(S ('D1 TRNNIn), Maybe (S ('D1 1)))]
parseJet = do
  nB <- A.double <* A.many1 A.space
  tracks <- A.many' . A.count 5 $ A.double <* A.many1 A.space

  return $ (, Just (S1D $ SA.konst nB)) . S1D . SA.fromList <$> tracks


-- can't export this....
-- trainRecurrentF
--   :: ( MonadRandom m
--      , Grenade.Recurrent.Core.Network.CreatableRecurrent layers shapes
--      , Num (RecurrentInputs layers)
--      , Data.Singletons.SingI (Last shapes)
--      )
--   => LearningParameters
--   -> F.FoldM
--      m
--      [(S (Head shapes), Maybe (S (Last shapes)))]
--      (RecurrentNetwork layers shapes, RecurrentInputs layers)
trainRecurrentF lps = F.FoldM step start done
  where
    start = randomRecurrent
    step (!rnet, !rins) samp =
      return $ trainRecurrent lps rnet rins samp
    done = return


runRecurrentP
  :: Monad m
  => RecurrentNetwork layers shapes
  -> RecurrentInputs layers
  -> Pipe [S (Head shapes)] [S (Last shapes)] m b
runRecurrentP net inps = do
  xs <- await
  let (_, _, ys) = runRecurrentNetwork net inps xs
  yield ys
  runRecurrentP net inps


main :: IO ()
main = do
    FeedForwardOpts nex rate <- execParser (info (feedForward' <**> helper) idm)
    putStrLn "Training network..."

    (net :: TracksNet, inps :: TracksInput) <-
      F.impurely P.foldM (trainRecurrentF rate)
      . over (chunksOf 1000) (\p -> liftIO (putStrLn "trained 1000 more") >> p)
      $ concats (view PT.lines PT.stdin)
        >-> P.map (A.maybeResult . A.parse parseJet)
        >-> P.concat
        >-> P.take nex

    print net

    -- let results = generateRecurrent trained bestInput (c 1)
    --
    -- print . take 50 . drop 100 $ results
