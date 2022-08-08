declare -a typs=("k" "g" "I" "kg" "kI" "gI" "kgI")
declare -a fields=("TT" "EE" "EB" "TE" "TB")
for typ in "${typs[@]}"
do
  echo "$typ"
  for field in "${fields[@]}"
  do
    echo "$field"
    python F_L.py 300 "$typ" S4 False "$field"
  done
done
declare -a fields=("EB" "TE" "TEB")
for typ in "${typs[@]}"
do
  echo "$typ"
  for field in "${fields[@]}"
  do
    echo "$field"
    python F_L.py 300 "$typ" S4 True "$field"
  done
done
