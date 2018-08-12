// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"

namespace Rivet {


  /// @brief Add a short analysis description here
  class BDisc : public Analysis {
  public:

    /// Constructor
    DEFAULT_RIVET_ANALYSIS_CTOR(BDisc);


    /// @name Analysis methods
    //@{

    /// Book histograms and initialise projections before the run
    void init() {

      // Initialise and register projections
      FinalState fps(Cuts::abseta < 5);

      declare(FastJets(fps, FastJets::ANTIKT, 0.4), "Jets");

    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {

      double weight = event.weight();

      const Jets& jets =
        apply<FastJets>(event, "Jets").jetsByPt(
            Cuts::pT > 25*GeV && Cuts::abseta < 2.5);

      if (jets.size() != 2)
        vetoEvent;

      for (const Jet& j: jets) {
        // if (!j.bTagged(Cuts::pT > 5*GeV))
            // continue;

        // printf("%.4e\t%.4e\t%.4e\t%.4e\n", weight, j.px(), j.py(), j.pz());
        printf("%d", j.bTags(Cuts::pT > 5*GeV).size());

        for (const Particle& c: j.constituents()) {
          if (!c.threeCharge())
            continue;

          // let's assume we can identify particles from b-hadron decays.
          bool fromB = c.fromBottom();
          printf("\t%d\t%.4e\t%.4e\t%.4e", fromB, c.px(), c.py(), c.pz());
        }

        cout << endl;
      }
    }


    /// Normalise histograms etc., after the run
    void finalize() {

    }

    //@}


    /// @name Histograms
    //@{
    Profile1DPtr _h_XXXX;
    Histo1DPtr _h_YYYY;
    CounterPtr _h_ZZZZ;
    //@}


  };


  // The hook for the plugin system
  DECLARE_RIVET_PLUGIN(BDisc);


}
