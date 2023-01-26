#declare -a arr=("kk" "gg" "gk" "kg" "II" "Ik" "kI" "gI" "Ig")
#declare -a arr=("II" "Ik" "kI" "gI" "Ig")
declare -a arr=("ss")
for typ in "${arr[@]}"
do
echo "$typ"
python M.py 5000 200 "True" "$typ"
#python M.py 10000 200 "False" "$typ"
done
