from run_experiments import run_all


def test_nn_classification():
    all_cae_paths = [
        # "C:\devl\work\ISH_Lasagne\src\DeepLearning\results_dae\CAE_16351_Shuffle_inputs-1502722116.38\run_31\\",
        # "C:\devl\work\ISH_Lasagne\src\DeepLearning\results_dae\CAE_16351_Shuffle_inputs-1502722116.38\run_88\\",
        # "C:\devl\work\ISH_Lasagne\src\DeepLearning\results_dae\CAE_16351_Shuffle_inputs-1502041138.18\run_18\\",
        # "C:\devl\work\ISH_Lasagne\src\DeepLearning\results_dae\CAE_16351_different_sizes-1495315103.29\run_100\\",
        "C:\devl\work\ISH_Lasagne\src\DeepLearning\results_dae\CAE_16351_Shuffle_inputs-1502722116.38\run_11\\"
    ]
    all_cae_sizes = [
        # 16,
        # 17,
        # 11,
        # 3,
        5
    ]
    for i in [0]:  # range(all_cae_paths.__len__()):
        LOAD_CAE_PATH = all_cae_paths[i].replace("\r", "\\r")
        print("Running NN classification for " + LOAD_CAE_PATH)
        run_all(use_nn_classifier=True, folder_name=LOAD_CAE_PATH, input_size_pre=all_cae_sizes[i])


if __name__ == "__main__":

    test_nn_classification()
