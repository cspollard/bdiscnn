{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections       #-}
{-# LANGUAGE TypeFamilies        #-}
{-# LANGUAGE TypeOperators       #-}

import qualified Control.Foldl                as F
import           Control.Lens
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
import qualified Pipes.Prelude.Text           as PT
import           System.IO                    (BufferMode (..), hSetBuffering,
                                               stdout)

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
  nB <- A.double
  tracks <- A.many1 . A.count 4 $ A.char '\t' *> A.double

  return $
    appLast Nothing (Just . S1D $ SA.konst nB)
    $ S1D . SA.fromList <$> tracks

  where
    -- appLast _ _ []     = []
    appLast _ y [z]    = [(z, y)]
    appLast x y (z:zs) = (z, x) : appLast x y zs


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
    hSetBuffering stdout LineBuffering

    putStrLn "Training network..."

    let p =
          PT.stdinLn
          >-> P.map (A.parseOnly parseJet)
          >-> P.concat

    (net :: TracksNet, inps :: TracksInput) <-
      F.impurely P.foldM (trainRecurrentF rate)
      . over (chunksOf 1000) (maps $ \x -> liftIO (putStrLn "trained 1000 more") >> x)
      $ p >-> P.take nex

    print net

    runEffect $
      p
      >-> P.mapM (\x -> print x >> return x)
      >-> P.map (fmap fst)
      >-> runRecurrentP net inps
      >-> P.print
