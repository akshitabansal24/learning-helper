1. create new email intellitutor@gmail.com ; pw: Akshita@2025
2. new project in gcp
3. add billing account
4. git clone https://github.com/akshitabansal24/learning-helper.git
5. git checkout feature/...
6. enable apis as per code error flow or directlt turn them on (Cloud Vision API, Vertex AI API, App Engine, datastore)
7. (maybe) import python libraries as per error
8. then run gcloud app deploy
9. make bucket public bucket->permissions->new principals = allUsers -> role = cloud storage = storage object viewer -> save
10. change projectId in script.js
11. To make database; go to -> firestore in gcp -> create database -> native mode -> databaseId = (default) -> test rules -> create -> collectionName = feedback -> save
12. go to firebase in gcp and check the db created there as well under same project
13. gcloud app deploy 
14. set location to us-central
15. target url(sample): https://learning-helper-2025-451212.uc.r.appspot.com/
16. python main.py for debugging
17. gcloud app logs tail -s default
