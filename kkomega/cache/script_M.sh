declare -a arr=("kk" "gg" "gk" "kg" "II" "Ik" "kI" "gI" "Ig")
#declare -a arr=("II" "Ik" "kI" "gI" "Ig")
for typ in "${arr[@]}"
do
echo "$typ"
python M.py 5000 200 "True" "$typ"
done
