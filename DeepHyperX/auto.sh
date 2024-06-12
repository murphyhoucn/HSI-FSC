python main.py --model SVM --cuda 0 --dataset Salinas 
rm -rf checkpoints/
python main.py --model SVM --cuda 0 --dataset IndianPines 
rm -rf checkpoints/
python main.py --model SVM --cuda 0 --dataset PaviaU 
rm -rf checkpoints/
# python main.py --model SVM --cuda 0 --dataset PaviaC 
# rm -rf checkpoints/
python main.py --model SVM --cuda 0 --dataset Xuzhou
rm -rf checkpoints/

python main.py --model nn --cuda 0 --dataset Salinas 
rm -rf checkpoints/
python main.py --model nn --cuda 0 --dataset IndianPines 
rm -rf checkpoints/
python main.py --model nn --cuda 0 --dataset PaviaU 
rm -rf checkpoints/
# python main.py --model nn --cuda 0 --dataset PaviaC 
# rm -rf checkpoints/
python main.py --model nn --cuda 0 --dataset Xuzhou 
rm -rf checkpoints/

python main.py --model hamida --cuda 0 --dataset Salinas 
rm -rf checkpoints/
python main.py --model hamida --cuda 0 --dataset IndianPines 
rm -rf checkpoints/
python main.py --model hamida --cuda 0 --dataset PaviaU 
rm -rf checkpoints/
# python main.py --model hamida --cuda 0 --dataset PaviaC 
# rm -rf checkpoints/
python main.py --model hamida --cuda 0 --dataset Xuzhou
rm -rf checkpoints/

python main.py --model luo --cuda 0 --dataset Salinas 
rm -rf checkpoints/
python main.py --model luo --cuda 0 --dataset IndianPines 
rm -rf checkpoints/
python main.py --model luo --cuda 0 --dataset PaviaU 
rm -rf checkpoints/
# python main.py --model luo --cuda 0 --dataset PaviaC
# rm -rf checkpoints/
python main.py --model luo --cuda 0 --dataset Xuzhou
rm -rf checkpoints/

python main.py --model mou --cuda 0 --dataset Salinas 
rm -rf checkpoints/
python main.py --model mou --cuda 0 --dataset IndianPines 
rm -rf checkpoints/
python main.py --model mou --cuda 0 --dataset PaviaU 
rm -rf checkpoints/
# python main.py --model mou --cuda 0 --dataset PaviaC 
# rm -rf checkpoints/
python main.py --model mou --cuda 0 --dataset Xuzhou
rm -rf checkpoints/