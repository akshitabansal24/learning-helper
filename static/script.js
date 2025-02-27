var recognizing = false;
var recognition = new webkitSpeechRecognition();
recognition.continuous = true;
var userAnswer="";
var correctAnswer="";

recognition.onend = function() {
    recognizing = false;
    document.querySelectorAll(".listen-button").forEach(btn => {
        btn.innerHTML = "🎤 Start Listening";
        btn.classList.remove("btn-danger");
        btn.classList.add("btn-success");
    });
    console.log("user ans:", userAnswer);
    checkAnswer(userAnswer, correctAnswer);
    userAnswer = "";
};

recognition.onresult = function(event) {
    for (var i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
            userAnswer+=event.results[i][0].transcript + " ";
        }
    }
};

function toggleStartStop(button,answer) {
    if (recognizing) {
        recognition.stop();
        console.log("correct ans:", answer);
        correctAnswer = answer;
    } else {
        recognition.start();
        recognizing = true;
        button.innerHTML = "🛑 Stop Listening";
        button.classList.remove("btn-success");
        button.classList.add("btn-danger");
    }
}

function uploadImage() {
    const fileInput = document.getElementById('fileInput');
    const cleanedImageBox = document.getElementById('cleanedImageBox');
    const qaBox = document.getElementById('qaBox');
    const uploadBtn = document.getElementById('uploadBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');

    if (!fileInput.files.length) {
        alert("Please select an image first.");
        return;
    }

    uploadBtn.disabled = true;
    loadingSpinner.classList.remove("d-none");

    const formData = new FormData();
    formData.append('img', fileInput.files[0]);

    fetch('https://learning-helper-2025-451212.uc.r.appspot.com/processImage', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(result => {
        if (result.cleaned_image_url) {
            cleanedImageBox.innerHTML = `<img src="${result.cleaned_image_url}" class="img-fluid rounded shadow" alt="Cleaned Image">`;
        } else {
            cleanedImageBox.innerHTML = `<p>No cleaned image received.</p>`;
        }

        if (result.qa_response && result.qa_response.QA && Array.isArray(result.qa_response.QA)) {
            qaBox.innerHTML = result.qa_response.QA.map((qa, index) => `
                <div class="alert alert-light border qa-card">
                    <strong>Q:</strong> ${qa.Ques}<br>
                    <strong>A:</strong> ${qa.Ans}
                    <div class="mt-2">
                        <button class="btn btn-sm btn-primary me-2" onclick="speakQues('${qa.Ques}')">
                            🔊 Play Question
                        </button>
                        <button id="listenBtn${index}" class="btn btn-sm btn-success listen-button" onclick="toggleStartStop(this, '${qa.Ans}')">
                            🎤 Start Listening
                        </button>
                    </div>
                </div>
            `).join('');
        } else {
            qaBox.innerHTML = "<p>No QA data found in response</p>";
        }
    })
    .catch(error => {
        console.error("Error:", error);
        qaBox.innerText = "Error processing the response";
    })
    .finally(() => {
        uploadBtn.disabled = false;
        loadingSpinner.classList.add("d-none");
    });
}

function speakQues(questionText) {
    const speechSynth = window.speechSynthesis;
    if (!speechSynth.speaking && questionText.trim().length) {
        const newUtter = new SpeechSynthesisUtterance(questionText);
        speechSynth.speak(newUtter);
    }
}

function checkAnswer(userAnswer, correctAnswer) {
    // const similarity = compareText(userAnswer.toLowerCase(), correctAnswer.toLowerCase());
    // console.log('Your answer similarity: ', similarity);
    var answers = JSON.stringify({'userAnswer': userAnswer,'correctAnswer': correctAnswer});

    fetch('https://learning-helper-2025-451212.uc.r.appspot.com/checkAnswer', {
        method: 'POST',
        headers: {
            'Accept': 'application/json, text/plain',
            'Content-Type': 'application/json;charset=UTF-8'
        },
        body: answers
    })
    .then(response => response.json())  
    .then(data => console.log(data));
}

function compareText(str1, str2) {
    let matches = 0;
    const words1 = str1.split(" ");
    const words2 = str2.split(" ");

    words1.forEach(word => {
        if (words2.includes(word)) matches++;
    });

    return ((matches / Math.max(words1.length, words2.length)) * 100).toFixed(2);
}