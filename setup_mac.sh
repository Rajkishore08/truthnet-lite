#!/bin/bash
echo "Setting up TruthNet Lite environment..."

# Check if brew is installed
if ! command -v brew &> /dev/null
then
    echo "Homebrew not found. Please install Homebrew first:"
    echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi

echo "Installing open-mpi..."
brew install open-mpi

echo "Creating Python virtual environment in backend/..."
mkdir -p backend frontend
cd backend
python3 -m venv truthnet-env

echo "Activating virtual environment and installing dependencies..."
source truthnet-env/bin/activate
pip install -r requirements.txt

# Download NLTK data
python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

echo "Backend setup complete."

echo "Setting up Frontend..."
cd ../frontend/truthnet-ui
npm install

echo "Setup complete! You can now start the frontend and backend."
