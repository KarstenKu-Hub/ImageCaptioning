# In this file you can run all the helper functions to get a better understanding of the input and output
# from them
# Inside the test_images and test_captions Directory you can find some test images for the functions below
# This are randomly picked pictures from the flickr8k Dataset

from data_loader import Vocabulary, CustomDataset, get_loader
import torchvision.transforms as transforms
import pandas as pd



if __name__ == "__main__":
    # How does the Vocabulary class for the tokenizing process work?
    print("")
    print("How does the Vocabulary class and the CustomDataset class for the tokenizing process work?")
    print("")
    # Load the data
    data_test = pd.read_csv("test_captions.txt")

    # Split the data to image(names) and captions
    images = data_test["image"]
    captions = data_test["caption"].tolist()    # change the type from pandas.core.series to list

    # Build the Vocabulary
    vocabulary = Vocabulary(freq_threshold=2)
    vocabulary.build_vocabulary(captions)

    example_sentence = "A black dog runs along the beach ."  # example sentence from test_captions.txt
    print("Example sentence: " + example_sentence)

    # First step: Tokenize the sentence
    example_sentence_tokenized = vocabulary.tokenizer_eng(example_sentence)
    print("Tokenized Example sentence: " + str(example_sentence_tokenized))

    # Second step: Numericalize the tokenized sentence (Step 1 ist already included in numericalize() )
    example_sentence_numericalized = vocabulary.numericalize(example_sentence)
    print("Numericalized Example sentence: " + str(example_sentence_numericalized))
    # -> This is the data format the network gets to start the training

    # How does the CustomDataset class work?
    # If you want to create a DataLoader for the training, you need to implement a class with two special functions
    # the __len__()- and the __getitem__(index)- functions.

    custom_dataset = CustomDataset("test_images", "test_captions.txt", freq_threshold=2)

    _, example_sentence_padded = custom_dataset.__getitem__(0)

    print("Numericalized Example sentence with start token and end token: " + str(example_sentence_padded))
    print("")


    # For batch_size = 3 you can see the padding process below via the pad_sequence function from MyCollate

    root_folder = "test_images"
    captions_file = "test_captions.txt"
    batch_size = 3
    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor(),]
    )


    loader, _ = get_loader(root_folder=root_folder, captions_file=captions_file, transform=transform, batch_size=batch_size)

    for idx, (imgs, captions) in enumerate(loader):
        captions = captions
        print("Shape of image data for first batch: " + str(imgs.shape))
        print("Shape of caption data fo first batch: " + str(captions.shape))
        print("")
        break

    example_sentence_padded = captions[:, 0]
    print("Random Padded sentence in the first batch of the DataLoader : " + str(example_sentence_padded))
    print("")























