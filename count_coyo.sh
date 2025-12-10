FILES=(/mnt/proj3/open-35-39/datasets/coyo_test/*.tar)
TOTAL_FILES=${#FILES[@]}
TOTAL=0
i=0

for f in "${FILES[@]}"; do
  COUNT=$(tar -tf "$f" | grep -E '\.(jpg|jpeg|png|webp)$' | wc -l)
  TOTAL=$((TOTAL + COUNT))

  i=$((i + 1))
  PERCENT=$((100 * i / TOTAL_FILES))
  printf "\rProgress: %d%% (%d/%d)" "$PERCENT" "$i" "$TOTAL_FILES"
done

echo
echo "Total images: $TOTAL"
