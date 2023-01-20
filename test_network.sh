source ./venv/bin/activate
#for batch_size in 32 64 128 256 512 1024
#for batch_size in 16 32 64 128 256
for batch_size in 32 128 256 512 1024
do
  python main.py -n "$1" --batch-size $batch_size --use-cuda --force-reload -m "$2"
  echo  "Done $1 with batch-size $batch_size"
done