#rm -rf bin*
mkdir bin
#CUDA_VISIBLE_DEVICES="0" python -m pdb extract_feature.py
CUDA_VISIBLE_DEVICES="1" python extract_feature.py
#CUDA_VISIBLE_DEVICES="0,1,2,3" python extract_feature.py
#CUDA_VISIBLE_DEVICES="4,5,6,7" python extract_feature.py

#tar -cf bin.tar bin/ 
#osscmd put bin.tar oss://jinzhiyong/pengyuli/bin.tar
