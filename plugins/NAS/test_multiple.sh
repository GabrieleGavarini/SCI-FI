export PYTHONPATH=~/PycharmProjects/FaultInjectionSpeedUp:$PYTHONPATH
#for cdsize in 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9
for cdsize in 3e-6 3e-7 5e-2 5e-3 5e-4 5e-5 5e-6 5e-7 5e-8 5e-9
do
   python main.py --cd-size $cdsize --batch-size 128 --use-cuda -m stuck-at_params --save-compressed
done