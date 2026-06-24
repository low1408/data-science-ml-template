from __future__ import annotations

from src import (
    Feature,
    FeatureColumns,
    FeaturePipeline,
    PreprocessingConfig,
    SQLiteDataLoader,
    run_pipeline,
)


def main() -> None:
    print("1. Loading raw dataset from SQLite database...")
    loader = SQLiteDataLoader("data/online_shopping.db")
    df = loader.load()
    print(f"Loaded DataFrame with shape: {df.shape}")

    # 2. Instantiate our FeaturePipeline with the custom feature
    print("\n2. Initializing custom feature engineering pipeline...")
    ratio_feature = Feature.from_fn(
        name="exit_bounce_ratio",
        requires=["ExitRate", "BounceRate"],
        fn=lambda df: df["ExitRate"] / (df["BounceRate"] + 1e-5),
        role="numeric",
    )
    feature_pipeline = FeaturePipeline([ratio_feature])

    # 3. Configure preprocessing roles
    # "exit_bounce_ratio" is registered through ratio_feature.role.
    feature_columns = FeatureColumns(
        numeric=[
            "SpecialDayProximity",
            "ExitRate",
            "BounceRate",
            "PageValue",
            "ProductPageTime",
        ],
        categorical=["CustomerType", "TrafficSource", "GeographicRegion"],
        boolean=[]
    )
    config = PreprocessingConfig(
        feature_columns=feature_columns,
        scale_numeric=True
    )

    # 4. Run the orchestrator
    print("\n3. Executing orchestrator pipeline...")
    result = run_pipeline(
        df,
        target_column="PurchaseCompleted",
        task="classification",
        config=config,
        feature_pipeline=feature_pipeline,
        test_size=0.2,
        stratify=True,
    )

    print("\nModel Comparison Results:")
    print(result.comparison)

    # 5. Verify inference works on raw unseen data (which does not contain the exit_bounce_ratio column)
    print("\n4. Verifying inference on new raw data...")
    raw_sample = df.drop(
        columns=["PurchaseCompleted", "exit_bounce_ratio"],
        errors="ignore",
    ).head(3)
    print("Input data to predict (raw columns only):")
    print(raw_sample[["ExitRate", "BounceRate", "CustomerType"]])

    model = result.models["logistic_regression"]
    predictions = model.predict(raw_sample)
    probabilities = model.predict_proba(raw_sample)[:, 1]

    print("\nPredictions:", predictions)
    print("Probabilities:", probabilities)
    print("\nIntegration test completed successfully!")


if __name__ == "__main__":
    main()
