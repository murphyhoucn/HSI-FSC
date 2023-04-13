@REM call python main.py --model SVM --cuda 0 --dataset Salinas 
@REM call python main.py --model SVM --cuda 0 --dataset IndianPines 
@REM call python main.py --model SVM --cuda 0 --dataset PaviaU 
@REM call python main.py --model SVM --cuda 0 --dataset PaviaC 
@REM call python main.py --model SVM --cuda 0 --dataset Xuzhou

call python main.py --model nn --cuda 0 --dataset Salinas 
call python main.py --model nn --cuda 0 --dataset IndianPines 
call python main.py --model nn --cuda 0 --dataset PaviaU 
call python main.py --model nn --cuda 0 --dataset PaviaC 
call python main.py --model nn --cuda 0 --dataset Xuzhou 


