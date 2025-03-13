from dateutil.parser import parse
from colorama import Fore, Style
from senti_ai.params import *
from senti_ai.ml_logic.data import load_data_from_bq
from senti_ai.ml_logic.registry import load_model, save_model, save_results
from senti_ai.ml_logic.model import initialize_model, train_model, compile_model, evaluate_model

def train(
        min_date:str = '2009-01-01',
        max_date:str = '2015-01-01',
        split_ratio: float = 0.02, # 0.02 represents ~ 1 month of validation data on a 2009-2015 train set
        learning_rate=0.0005,
        batch_size = 256,
        patience = 2
    ) -> float:


    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading data..." + Style.RESET_ALL)

    #min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    #max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    df = load_data_from_bq(GCP_PROJECT,BQ_DATASET,"raw")
    df = df.sort_values("date", ascending=False)

    X_train = df.drop('BTC_Close',axis=1)
    y_train = df[['BTC_Close']]

    model = load_model()

    if model is None:
        model = initialize_model(input_shape=X_train.shape[1:])

    model = compile_model(model, learning_rate=learning_rate)
    model, history = train_model(
        model, X_train, y_train,
        batch_size=batch_size,
        patience=patience,
        validation_split=0.2
    )

    val_mae = np.min(history.history['val_mae'])

    params = dict(
        context="train",
        #training_set_size=DATA_SIZE,
        row_count=len(X_train),
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(mae=val_mae))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    # The latest model should be moved to staging
    #mlflow_transition_model('None', 'Staging')

    print("✅ train() done \n")

    return val_mae
