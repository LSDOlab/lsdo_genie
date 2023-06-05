# Developer Guidelines

## Pull Requests
Before contributing to the `main` branch, make sure the following requirements are met:
- [ ] All existing tests have passed
- [ ] No local merge conflicts
- [ ] Code is commented
- [ ] Added tests for new features

Once you finish making changes on your branch, create a pull request.
To create the pull request, follow the steps:

1. Pull from `main` branch
```sh
git pull # Get most up-to-update version
```
2. Merge with main and check for conflicts
```sh
git merge main # merge with main locally on your branch to check for conflicts
```
3. Run tests
```sh
pytest # standard testing
pytest --disable-warnings # tests without displaying warnings
pytest -rP # tests while displaying print statements
```

4. Push changes

```sh
git push
```

5. Create pull request on GitHub with descriptions for changes.
 - Fill out pull request template
 - Assign reviewers (repo admin(s)?)
