"""Network-condition batch entry point (fixed depth, varying j).

Runs all dataset queries at a single fixed research depth (DEPTH_VALUE=2),
so that poisoning impact can be measured as a function of the number of
injected documents j rather than research depth.  Results are archived under
outputs/batch_web_poison_network/.

For the depth-ablation experiment (paper Figure 3), use experiments/run_depth.py
instead.
"""

from tools.run_network_experiment import main


if __name__ == "__main__":
    main()
