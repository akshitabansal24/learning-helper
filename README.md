1. create new email intellitutor@gmail.com ; pw: same gmail pw
2. new project in gcp
3. add billing account
4. git clone https://github.com/akshitabansal24/learning-helper.git
5. git checkout feature/
6. enable apis as per code error flow (Cloud Vision API, Vertex AI API, App Engine, datastore)
7. (maybe) import python libraries as per error
8. make bucket public bucket->permissions->new principals = allUsers -> role = cloud storage = storage object viewer -> save
9. change projectId in script.js
10. To make database; go to -> firestore in gcp -> create database -> native mode -> databaseId = (default) -> test rules -> create -> collectionName = feedback -> save
11. go to firebase and check the db created there as well under same project
12. gcloud app deploy 
13. set location to us-central
14. target url(sample): https://learning-helper-2025-451212.uc.r.appspot.com/
15. python main.py for debugging
16. gcloud app logs tail -s default
