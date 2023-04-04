from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, HinSAGE, link_classification
from tensorflow import keras


class GraphSage:

    def __init__(self, epochs, batch_size, num_samples, layer_size) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.layer_sizes = layer_size

    def prepare_data(self, G_train, edge_ids_train, edge_labels_train, G_test, edge_ids_test, edge_labels_test):
        self.train_gen = GraphSAGELinkGenerator(G_train, self.batch_size, self.num_samples)
        self.train_flow = self.train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)
        self.test_gen = GraphSAGELinkGenerator(G_test, self.batch_size, self.num_samples)
        self.test_flow = self.test_gen.flow(edge_ids_test, edge_labels_test)

    def build_model(self):
        graphsage = GraphSAGE(
            layer_sizes=self.layer_sizes, generator=self.train_gen, bias=True, dropout=0.3
        )
        x_inp, x_out = graphsage.in_out_tensors()
        prediction = link_classification(
            output_dim=1, output_act="relu", edge_embedding_method="ip")(x_out)
        model = keras.Model(inputs=x_inp, outputs=prediction)

        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss=keras.losses.binary_crossentropy,
            metrics=["acc"],
        )
        return model

    def model_fit(self, model, train_flow, test_flow):
        history = model.fit(train_flow, epochs=self.epochs, validation_data=test_flow, verbose=2)
        return model


    def evaluate(self, model, test_flow):
        test_metrics = model.evaluate(test_flow)
        print("\nTest Set Metrics of the trained model:")
        for name, val in zip(model.metrics_names, test_metrics):
            print("\t{}: {:0.4f}".format(name, val))
