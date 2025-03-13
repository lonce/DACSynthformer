import requests
from datetime import datetime 

OWNER = "lonce"
REPO = "DACSynthformer"
GITHUB_API = f"https://api.github.com/repos/{OWNER}/{REPO}"

# Fetch all branches
branches_url = f"{GITHUB_API}/branches"
branches = requests.get(branches_url).json()

commit_list = []

for branch in branches:
    branch_name = branch["name"]

    # Fetch commits for each branch (including merges)
    commits_url = f"{GITHUB_API}/commits?sha={branch_name}&per_page=50"  # Increase per_page as needed
    commits = requests.get(commits_url).json()

    for commit in commits:
        sha = commit["sha"][:7]  # Shortened SHA
        author = commit["commit"]["author"]["name"]
        message = commit["commit"]["message"]
        date = commit["commit"]["author"]["date"]  # ISO format: YYYY-MM-DDTHH:MM:SSZ
        branch_info = branch_name

        # Store in a list as tuple
        commit_list.append((date, sha, author, message, branch_info))

# Sort commits in **reverse chronological order**
commit_list.sort(key=lambda x: datetime.strptime(x[0], "%Y-%m-%dT%H:%M:%SZ"), reverse=True)

# Print results
print("\nRecent commits across all branches (newest first):")
for date, sha, author, message, branch_info in commit_list:
    print(f"{date} | {sha} - {author}: {message} [{branch_info}]")