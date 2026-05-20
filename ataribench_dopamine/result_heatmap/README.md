# Standalone Result Heatmap

This directory is self-contained. It stores the frozen comparison values for
the existing LLM rows and the Sample Factory APPO RL baseline in:

```text
baseline_scores.csv
```

After Dopamine eval results are written under:

```text
../../results/dopamine_tf_dqn/
```

generate the overall raw-score heatmap from the Dopamine repo root:

```bash
cd /Users/mingjiahuo/Desktop/ataribench/rl_baselines/dopamine
python ataribench_dopamine/result_heatmap/plot_heatmap.py
```

Default output:

```text
ataribench_dopamine/result_heatmap/plot_heatmap_raw_with_dopamine.png
```

`evaluate_all_30s.py` also calls this script automatically after batch eval:

```bash
python ataribench_dopamine/evaluate_all_30s.py
```

Use `--no-plot-heatmap` to skip plotting.
