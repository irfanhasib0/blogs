#!/bin/bash
npm run build
cp -r build/* ../docs/
cd .. 
git add .
git commit -m "Saving Update"
git push origin master
