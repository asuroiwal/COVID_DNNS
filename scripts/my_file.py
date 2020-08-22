class BalancedDataGenerator(tf.keras.utils.Sequence):
    def __init__(
            self,
            data_dir,
            df,
            x_col,
            y_col,
            is_training,
            batch_size,
            input_shape=(224,224),
            num_classes=3,
            num_channels=3,
            mapping=None,
            shuffle=True,
            augmentation=apply_augmentation,
            covid_percent=0.3,
            top_percent=0.08
    ):
        'Initialisation'
        if mapping is None:
            mapping = {
                'normal': 0,
                'pneumonia': 1,
                'COVID-19': 2
            }
        self.data_dir=data_dir,
        self.df=df,
        self.x_col=x_col,
        self.y_col=y_col,
        self.is_training=is_training,
        self.batch_size=batch_size,
        self.input_shape=input_shape,
        self.num_classes=num_classes,
        self.num_channels=num_channels,
        self.shuffle=shuffle,
        self.augmentation=augmentation,
        self.covid_percent=covid_percent,
        self.top_percent=top_percent
        datasets = {'normal': [], 'pneumonia': [], 'COVID-19': []}
        for l in self.dataset:
            datasets[l.split()[2]].append(l)
        self.datasets = [
            datasets['normal'] + datasets['pneumonia'],
            datasets['COVID-19'],
        ]
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]

        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X =  # logic
        y =  # logic

        for i, id in enumerate(batch):
            X[i,] =  # logic
            y[i] =  # labels

        return X, y