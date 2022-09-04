# from lazypredict.Supervised import LazyRegressor
import pickle

from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
from Model_DeepLearning import *
from UtilFunctions import *
from sklearn.metrics import median_absolute_error
def get_model(model, model_params):
    models = {
        "rnn": RNNModel,
        "lstm": LSTMModel,
        "gru": GRUModel,
    }
    return models.get(model.lower())(**model_params)


class Optimization:
    """Optimization is a helper class that allows training, validation, prediction.

    Optimization is a helper class that takes model, loss function, optimizer function
    learning scheduler (optional), early stopping (optional) as inputs. In return, it
    provides a framework to train and validate the models, and to predict future values
    based on the models.

    Attributes:
        model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
        loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
        optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
        train_losses (list[float]): The loss values from the training
        val_losses (list[float]): The loss values from the validation
        last_epoch (int): The number of epochs that the models is trained
    """

    def __init__(self, model, loss_fn, optimizer):
        """
        Args:
            model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
            loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
            optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):
        """The method train_step completes one step of training.

        Given the features (x) and the target values (y) tensors, the method completes
        one step of the training. First, it activates the train mode to enable back prop.
        After generating predicted values (yhat) by doing forward propagation, it calculates
        the losses by using the loss function. Then, it computes the gradients by doing
        back propagation and updates the weights by calling step() function.

        Args:
            x (torch.Tensor): Tensor for features to train one step
            y (torch.Tensor): Tensor for target values to calculate losses

        """
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1,fopModelPath='log/',device=None):
        """The method train performs the model training

        The method takes DataLoaders for training and validation datasets, batch size for
        mini-batch training, number of epochs to train, and number of features as inputs.
        Then, it carries out the training by iteratively calling the method train_step for
        n_epochs times. If early stopping is enabled, then it  checks the stopping condition
        to decide whether the training needs to halt before n_epochs steps. Finally, it saves
        the model in a designated file path.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader that stores training data
            val_loader (torch.utils.data.DataLoader): DataLoader that stores validation data
            batch_size (int): Batch size for mini-batch training
            n_epochs (int): Number of epochs, i.e., train steps, to train
            n_features (int): Number of feature columns

        """
        createDirIfNotExist(fopModelPath)
        model_path = '{}/{}'.format(fopModelPath,'currentModel.pkl')

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                # print('type {} {}'.format(type(x_batch),type(y_batch)))
                # input('abc')
                x_batch = x_batch.view([batch_size, -1, n_features]).cuda()
                y_batch = y_batch.cuda()
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)


            print(f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}")

        torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        """The method evaluate performs the model evaluation

        The method takes DataLoaders for the test dataset, batch size for mini-batch testing,
        and number of features as inputs. Similar to the model validation, it iteratively
        predicts the target values and calculates losses. Then, it returns two lists that
        hold the predictions and the actual values.

        Note:
            This method assumes that the prediction from the previous step is available at
            the time of the prediction, and only does one-step prediction into the future.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader that stores test data
            batch_size (int): Batch size for mini-batch training
            n_features (int): Number of feature columns

        Returns:
            list[float]: The values predicted by the model
            list[float]: The actual values in the test set.

        """
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().cpu().numpy())
                values.append(y_test.to(device).detach().cpu().numpy())

        return predictions, values

    def plot_losses(self):
        """The method plots the calculated loss values for training and validation
        """
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


def format_predictions(predictions, values, df_test, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    # print(preds)
    # print('type df_test {}'.format(type(df_test)))
    lstTestHeads=[i for i in range(0,len(df_test))]
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=lstTestHeads)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
    return df_result
def format_predictions_2(predictions, values, lstTestHeads, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    # print(preds)
    # print('type df_test {}'.format(type(df_test)))
    # lstTestHeads=[i for i in range(0,len(df_test))]
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=lstTestHeads)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
    return df_result


def calculate_metrics(df):
    result_metrics = {'mae': mean_absolute_error(df.value, df.prediction),
                      'rmse': mean_squared_error(df.value, df.prediction) ** 0.5,
                      'r2': r2_score(df.value, df.prediction)}

    print("Mean Absolute Error:       ", result_metrics["mae"])
    print("Root Mean Squared Error:   ", result_metrics["rmse"])
    print("R^2 Score:                 ", result_metrics["r2"])
    return result_metrics
def calculate_metrics_tolist(df):
    # result_metrics = {'mae': mean_absolute_error(df.value, df.prediction),
    #                   'rmse': mean_squared_error(df.value, df.prediction) ** 0.5,
    #                   'r2': r2_score(df.value, df.prediction)}
    #
    # print("Mean Absolute Error:       ", result_metrics["mae"])
    # print("Root Mean Squared Error:   ", result_metrics["rmse"])
    # print("R^2 Score:                 ", result_metrics["r2"])
    rmse=mean_squared_error(df.value, df.prediction) ** 0.5
    mse=mean_squared_error(df.value, df.prediction)
    mae=mean_absolute_error(df.value, df.prediction)
    mdae=median_absolute_error(df.value, df.prediction)
    result_metrics=[rmse,mse,mae,mdae]
    return result_metrics

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()

# result_metrics = calculate_metrics(df_result)


# reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
# models, predictions = reg.fit(X_train, X_test, y_train, y_test)
# print(models)
from torch.utils.data import TensorDataset, DataLoader




def runningDeepLearningModel(modelName,fopModelPath, X_train, X_valid, X_test, y_train, y_valid, y_test):
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    scaler = get_scaler('minmax')

    X_train_arr = scaler.fit_transform(X_train)
    pickle.dump(scaler, open(fopModelPath + 'scaler_X.pkl', 'wb'))

    X_val_arr = scaler.transform(X_valid)
    X_test_arr = scaler.transform(X_test)

    y_train=y_train.reshape(-1, 1)
    y_valid=y_valid.reshape(-1, 1)
    y_test=y_test.reshape(-1, 1)
    y_train_arr = scaler.fit_transform(y_train)
    pickle.dump(scaler, open(fopModelPath + 'scaler_y.pkl', 'wb'))
    y_val_arr = scaler.transform(y_valid)
    y_test_arr = scaler.transform(y_test)
    # print(y_valid)
    # print('compare between y_test and y_test_arr')
    # print('y_test : {}'.format(y_test))
    # print('y_test_arr : {}'.format(y_test_arr))
    # input('aaaa ')

    train_features = torch.Tensor(X_train_arr).cuda()
    train_targets = torch.Tensor(y_train_arr).cuda()
    val_features = torch.Tensor(X_val_arr).cuda()
    val_targets = torch.Tensor(y_val_arr).cuda()
    test_features = torch.Tensor(X_test_arr).cuda()
    test_targets = torch.Tensor(y_test_arr).cuda()

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)


    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)


    import torch.optim as optim

    input_dim = len(X_train[0])
    output_dim = 1
    hidden_dim = 256
    layer_dim = 3
    batch_size = 64
    dropout = 0.002
    n_epochs = 200
    learning_rate = 1e-3
    weight_decay = 1e-6

    model_params = {'input_dim': input_dim,
                    'hidden_dim' : hidden_dim,
                    'layer_dim' : layer_dim,
                    'output_dim' : output_dim,
                    'dropout_prob' : dropout}
    # input('aaa ')

    model = get_model(modelName, model_params)
    # model.to(device)
    model=model.cuda()

    loss_fn = nn.MSELoss(reduction="mean")
    # loss_fn.to(device)
    loss_fn=loss_fn.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
    # fopModelPath='log/'

    # print(device)
    # input('aaaa ')
    opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim,fopModelPath=fopModelPath,device=device)
    # opt.plot_losses()

    predictions, values = opt.evaluate(
        test_loader_one,
        batch_size=1,
        n_features=input_dim
    )
    createDirIfNotExist(fopModelPath)
    # pickle.dump(scaler,open(fopModelPath+'scaler_y.pkl','wb'))
    df_result = format_predictions(predictions, values, X_test, scaler)
    result_metrics = calculate_metrics_tolist(df_result)
    return model,df_result,result_metrics
    # print('result: {}'.format(result_metrics))
    # print('prediction: {}'.format(predictions))
    # print('dfresult: {}'.format(df_result))
# boston = datasets.load_boston()
# X, y = shuffle(boston.data, boston.target, random_state=13)
# X = X.astype(np.float32)
#
# offsetTest = int(X.shape[0] * 0.9)
# X_trainAndValid, y_trainAndValid = X[:offsetTest], y[:offsetTest]
# X_test, y_test = X[offsetTest:], y[offsetTest:]
# offsetValid=int(X_trainAndValid.shape[0] * 0.9)
# X_train,y_train=X_trainAndValid[:offsetValid],y[:offsetValid]
# X_valid,y_valid=X_trainAndValid[offsetValid:],y_trainAndValid[offsetValid:]
# modelDeep='rnn'
# model,df_result=runningDeepLearningModel(modelDeep,'log/',X_train,X_valid,X_test,y_train,y_valid,y_test)
# import pickle
# pickle.dump(model,open('model.bin','wb'))
