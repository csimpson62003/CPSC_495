"""
Google Colab Setup Script
=========================
Run this FIRST in Google Colab to set up git credentials.
This allows automatic checkpoint pushing to GitHub.

Usage:
1. Get a GitHub Personal Access Token:
   - Go to https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Select "repo" scope
   - Copy the token

2. Run this script in Colab:
   !python setup_colab.py

3. Enter your GitHub username and token when prompted

4. Then run: !python train_colab.py
"""

import subprocess
import os


def setup_git_credentials():
    """Configure git with user credentials for pushing to GitHub."""
    print("=" * 60)
    print("🔧 Git Configuration Setup")
    print("=" * 60)
    
    # Check if git is installed
    try:
        subprocess.run(['git', '--version'], check=True, capture_output=True)
        print("✅ Git is installed")
    except:
        print("❌ Git is not installed. Installing...")
        subprocess.run(['apt-get', 'install', '-y', 'git'], check=True)
    
    # Get user input
    print("\n📋 Please provide your GitHub credentials:")
    username = input("GitHub Username: ").strip()
    email = input("GitHub Email: ").strip()
    token = input("GitHub Personal Access Token: ").strip()
    
    if not username or not email or not token:
        print("❌ Error: All fields are required!")
        return False
    
    # Configure git
    print("\n⚙️  Configuring git...")
    subprocess.run(['git', 'config', '--global', 'user.name', username], check=True)
    subprocess.run(['git', 'config', '--global', 'user.email', email], check=True)
    
    # Store credentials
    subprocess.run(['git', 'config', '--global', 'credential.helper', 'store'], check=True)
    
    # Get current repository URL and update it with token
    try:
        result = subprocess.run(['git', 'remote', 'get-url', 'origin'], 
                              capture_output=True, text=True, check=True)
        repo_url = result.stdout.strip()
        
        # Parse repo URL
        if 'github.com' in repo_url:
            # Extract repo path (e.g., csimpson62003/CPSC_495)
            repo_path = repo_url.split('github.com/')[-1].replace('.git', '')
            
            # Create authenticated URL
            auth_url = f'https://{username}:{token}@github.com/{repo_path}.git'
            
            # Update remote URL
            subprocess.run(['git', 'remote', 'set-url', 'origin', auth_url], check=True)
            
            print("✅ Git credentials configured successfully!")
            print(f"   Repository: {repo_path}")
            print("\n🎉 You're all set! Run: !python train_colab.py")
            print("   Checkpoints will be automatically pushed to GitHub every 10 epochs.")
            return True
        else:
            print("⚠️  Warning: Not a GitHub repository")
            return False
            
    except subprocess.CalledProcessError:
        print("⚠️  Warning: Could not update remote URL")
        print("   You may need to manually configure git push")
        return False


def main():
    print("🚀 Google Colab - GitHub Setup")
    print()
    print("This script will configure git to automatically push")
    print("training checkpoints to your GitHub repository.")
    print()
    print("💡 Benefits:")
    print("   - Protects training progress if Colab disconnects")
    print("   - Automatically saves checkpoints every 10 epochs")
    print("   - No manual uploading needed!")
    print()
    
    setup_git_credentials()


if __name__ == "__main__":
    main()
