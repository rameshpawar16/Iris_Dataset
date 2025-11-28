from src.model import IrisModel, train_and_save_model
from pathlib import Path
import joblib


def test_model_training():
    model = IrisModel()
    model.train()

    assert hasattr(model, "predict")


def test_model_prediction():
    model = IrisModel()
    model.train()

    
    sample1 = [5.1, 3.5, 1.4, 0.2]
    sample2 = [5.1,3.5,1.4,0.2]
    sample3 = [4.9,3,1.4,0.2]
    sample4 = [4.7,3.2,1.3,0.2]
    sample5 = [4.6,3.1,1.5,0.2]

    assert model.predict(sample1) == "Setosa"
    assert model.predict(sample2) == "Setosa"
    assert model.predict(sample3) == "Setosa"
    assert model.predict(sample4) == "Setosa"
    assert model.predict(sample5) == "Setosa"

    v1 =[ 7,3.2,4.7,1.4]
    v2 =[6.4,3.2,4.5,1.5]	
    v3 =[6.9,3.1,4.9,1.5]
    v4 =[5.5,2.3,4,1.3]	

    assert model.predict(v1) == "Versicolor"
    assert model.predict(v2) == "Versicolor"
    assert model.predict(v3) == "Versicolor"
    assert model.predict(v4) == "Versicolor"

    a1 = [6.3,2.5,5,1.9]
    a2 = [6.5,3,5.2,2]
    a3 = [6.2,3.4,5.4,2.3]
    a4 = [5.9,3,5.1,1.8]
    
    assert model.predict(a1) == 'Virginica'
    assert model.predict(a2) == 'Virginica'
    assert model.predict(a3) == 'Virginica'
    assert model.predict(a4) == 'Virginica'


def test_model_file_saved(tmp_path):
    model_file = tmp_path / "data.pkl"

    train_and_save_model(model_file)

    assert Path(model_file).exists()

    loaded_model = joblib.load(model_file)
    assert hasattr(loaded_model, "predict")
