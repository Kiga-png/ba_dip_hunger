

echo "Metadata"
cd metadata
python metadata.py
cd ..

echo "Spot potential repeats"
cd spot_repeats
python spot_repeats.py
cd ..

echo "FINISHED"
