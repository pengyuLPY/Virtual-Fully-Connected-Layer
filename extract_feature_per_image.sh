#rm -rf lfw_predict lfw_predict.tar
#mkdir lfw_predict
#CUDA_VISIBLE_DEVICES="0" python -m pdb extract_feature_per_image.py
CUDA_VISIBLE_DEVICES="7" python extract_feature_per_image.py
#tar -cf lfw_predict.tar lfw_predict
#osscmd put lfw_predict.tar oss://jinzhiyong/pengyuli/lfw_predict.tar

