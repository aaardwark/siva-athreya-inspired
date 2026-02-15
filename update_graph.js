function update_graph(updates) {
    const updatesJSON = JSON.parse(updates);
    for (style in updatesJSON) {
        console.log(updatesJSON[style]);
        var counter = 0;
        for (id of updatesJSON[style]) {
            document.getElementById(id).setAttribute('class', style);
            counter += 1;
        }
        console.log(counter);
    }   
}