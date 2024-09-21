from zenml.client import Client

from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    # Run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="data/olist_customers_dataset.csv")

# mlflow ui --backend-store-uri "file:C:\Users\hp\AppData\Roaming\zenml\local_stores\b3ade5fd-8d90-4ba2-ac54-d6b832873e68\mlruns"