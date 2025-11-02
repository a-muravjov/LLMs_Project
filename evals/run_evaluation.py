from evaluate_all import (
    summarize_results,
    analyze_dataset,
    plot_per_emotion_heatmap,
    plot_metrics_by_model,
    plot_lora_vs_qlora
)

if __name__ == "__main__":
    df = summarize_results()

    # change path to analyze any other dataset's label distribution
    analyze_dataset("datasets/EN/train.json")

    plot_per_emotion_heatmap(df)
    plot_metrics_by_model(df)
    plot_lora_vs_qlora(df, metric="Jaccard")
