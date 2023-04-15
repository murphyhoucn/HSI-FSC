call python test.py --datasetname SA --count 0
call python test.py --datasetname SA --count 1
call python test.py --datasetname SA --count 2
call python test.py --datasetname SA --count 3
call python test.py --datasetname SA --count 4
call python test.py --datasetname SA --count 5
call python test.py --datasetname SA --count 6
call python test.py --datasetname SA --count 7
call python test.py --datasetname SA --count 8
call python test.py --datasetname SA --count 9

@REM 在 RTX A4000, 32GB服务器上, SA测试程序被kill了! 应该是OOM, 所以只能在windows上再跑一次。 