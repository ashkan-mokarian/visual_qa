- data_provider: provides train_set, validation_set, test_set, and dictionary for further preperations, like pruning and tranforming to tensors and ... . Must provide either a path which already has preprocessed data as text_data.pkl and dict.pkl or a default preprocessor and the name of it e.g. daquar_only_text. main function to call is load_data

- data_prepare: contains different preperation steps such as to Tensor transformation, pruning sample answers to only single words and etc.
