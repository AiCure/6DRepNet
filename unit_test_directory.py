from sixdrepnet.sixdrepnet import process_directory


if __name__ == '__main__':
    process_directory('./test_videos', './processed_data', num_threads=1)