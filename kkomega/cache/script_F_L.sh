declare -a typs=("k" "g" "I" "kg" "kI" "gI" "kgI")
for typ in "${typs[@]}"
do
echo "$typ"
python F_L.py 300 "$typ" S4
done
