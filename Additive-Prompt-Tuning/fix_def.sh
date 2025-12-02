#!/bin/bash
# Script to fix validation method in learners/default.py

echo "Fixing learners/default.py to use train=False in validation..."

# Backup original
cp learners/default.py learners/default.py.backup

# Fix line 201: add train=False
sed -i '201s/model.forward(input)/model.forward(input, train=False)/' learners/default.py

# Fix line 213: add train=False (if exists)
sed -i '213s/model.forward(input,local_test=False)/model.forward(input, train=False, local_test=False)/' learners/default.py

# Fix line 216: add train=False (if exists)
sed -i '216s/model.forward(input,local_test=True)/model.forward(input, train=False, local_test=True)/' learners/default.py

echo "Done! Backup saved as learners/default.py.backup"
echo ""
echo "Changes made:"
echo "  Line 201: Added train=False to validation"
echo "  Line 213: Added train=False to global test"
echo "  Line 216: Added train=False to local test"