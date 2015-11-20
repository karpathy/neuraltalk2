# will spawn workers on the given GPU ids, in screen sessions prefixed with "ak"
for i in "$@"
do

  echo "spawning worker on GPU $i..."
  screen -S ak$i -d -m ./runworker.sh $i

  sleep 2
done


