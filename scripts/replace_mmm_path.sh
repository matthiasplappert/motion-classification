grep -rl 'mmm.xml' ./ | xargs sed -i "" 's/mmm.xml/..\/model\/mmm.xml/g'
