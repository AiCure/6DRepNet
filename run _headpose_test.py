
import os
import run_headpose as headpose
from  batch_base.testing_util import clean_up_s3_output, check_s3_items

def test_openface(s3_path_to_input_csv, 
                s3_path_to_result,
                s3_path_to_output_prefix,
                s3_path_to_model):
    print("Test headpose face")
    
    headpose.run_headpose(
                s3_path_to_input_csv, 
                s3_path_to_result,
                s3_path_to_output_prefix,
                s3_path_to_model
    )
    # can write a check for the output you expect below

    # out_path = s3_path_to_output_prefix + var1 + os.path.dirname('/'.join(data_id.split('/')[:-1])) + '/'
    # assert('dbm_1video_querycolumns.csv.output_paths' in check_s3_items(out_path))
    # out_path = s3_path_to_output_prefix + var2 + os.path.dirname('/'.join(data_id.split('/')[:-1])) + '/'
    # assert('dbm_1video_querycolumns.csv.output_paths' in check_s3_items(out_path))

if __name__ == '__main__':
    s3_path_to_input_csv = 's3://simulated-td-videos/dev_encrtyped_videos.csv' 
    s3_path_to_result = 's3://simulated-td-videos/encrypted_videos_result'
    s3_path_to_output_prefix = 's3://simulated-td-videos/encrypted_videos_result/'
    s3_path_to_model = 's3://simulated-td-videos/6DRepNet_300W_LP_AFLW2000.pth'
    test_openface(s3_path_to_input_csv, 
                s3_path_to_result,
                s3_path_to_output_prefix,
                s3_path_to_model)
    # if you checked for your expected output use this command to clean the s3 bucket

    # clean_up_s3_output(s3_path_to_output_prefix + var1)
    # clean_up_s3_output(s3_path_to_output_prefix + var2)