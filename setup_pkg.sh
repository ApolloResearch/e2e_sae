#!/bin/bash

# This script does the following:
# 1. Replaces all instances of the string 'mypkg' with the first argument provided
# 2. Renames the mypkg directory to the first argument provided
# 3. Copies the pre-commit file to .git/hooks/pre-commit and makes it executable

if [ $# -eq 0 ]; then
    echo "No arguments provided. Please provide the new package name."
    exit 1
fi

# Get the new package name from the first argument
new_package_name=$1

OS=$(uname)

# List of file types
file_types=("*.md" "*.py" "*.yaml" "*.json")

# Loop over each file type
for file_type in "${file_types[@]}"; do
    # Use find to get a list of all matching files in the current directory and subdirectories
    if [[ "$OS" == "Darwin" ]]; then # If the operating system is MacOS
        find . -type f -name "$file_type" -print0 | xargs -0 sed -i '' "s/mypkg/$new_package_name/g"
    else # Assume GNU/Linux otherwise
        find . -type f -name "$file_type" -print0 | xargs -0 sed -i "s/mypkg/$new_package_name/g"
    fi
done

# Also rename the mypkg directory
mv mypkg $new_package_name

# Use the provided pre-commit file
cp .pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit