from main.celery import app

@app.task()
def dataset_preparation(dataset_id:int):
    print(f"{dataset_id=}")
