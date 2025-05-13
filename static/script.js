var recognizing = false;
var recognition = new webkitSpeechRecognition();
recognition.continuous = true;
var questions=[];
var userAnswer=[];
var correctAnswer=[];
var feedback=[];
var pdfResponse = "";
var cleanedImageUrl = "";
var speakIndex = "";
var projectId = "intellitutor-2025";

recognition.onend = function() {
    recognizing = false;
    document.querySelectorAll(".listen-button").forEach(btn => {
        btn.innerHTML = "<i class='bi bi-mic'></i> Start Listening";
    });
    const textArea = document.getElementById('textArea'+speakIndex);
    textArea.value = userAnswer[speakIndex];
    userAnswer[speakIndex] = "";
};

recognition.onresult = function(event) {
    for (var i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
            userAnswer[speakIndex]+=event.results[i][0].transcript + " ";
        }
    }
};

function toggleStartStop(button,index) {
    speakIndex = index;
    if (recognizing) {
        recognition.stop();
    } else {
        recognition.start();
        recognizing = true;
        button.innerHTML = "<i class='bi bi-mic-mute'></i> Stop Listening";
    }
}

function uploadImage(quizMode) {
    const fileInput = document.getElementById('fileInput');
    const cleanedImageBox = document.getElementById('cleanedImageBox');
    cleanedImageBox.innerHTML = "";
    const qaBox = document.getElementById('qaBox');
    qaBox.innerHTML = "";
    const loadingSpinner = document.getElementById('loadingSpinner');
    const responseScreen = document.getElementById('responseScreen');
    document.getElementById('cleanedImageDiv').classList.remove("d-none");
    document.getElementById('QADiv').classList.remove('col-md-12');
    document.getElementById('QADiv').classList.add('col-md-6');
    document.getElementById('uploadFeedback_btn').classList.remove("d-none");
    questions=[];
    userAnswer=[];
    correctAnswer=[];
    feedback=[];
    pdfResponse = "";
    cleanedImageUrl = "";
    speakIndex = "";

    if (!fileInput.files.length) {
        alert("Please select an image first.");
        return;
    }

    uploadBtn_study.disabled = true;
    uploadBtn_quiz.disabled = true;
    loadingSpinner.classList.remove("d-none");

    const formData = new FormData();
    formData.append('img', fileInput.files[0]);

    fetch('https://'+projectId+'.uc.r.appspot.com/processImage', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(result => {
        if (result.cleaned_image_url) {
            cleanedImageUrl = result.cleaned_image_url;
            cleanedImageBox.innerHTML = `<img src="${result.cleaned_image_url}" class="img-fluid rounded shadow" alt="Cleaned Image">`;
        } else {
            cleanedImageBox.innerHTML = `<p>No cleaned image received.</p>`;
        }
        console.log(result);
        if (result.qa_response && Array.isArray(result.qa_response)) {
            pdfResponse = result.qa_response;
            result.qa_response.forEach(element => {
                questions.push(element.Ques);
                correctAnswer.push(element.Ans);
                userAnswer.push("");
                feedback.push({});
            });
            qaBox.innerHTML = result.qa_response.map((qa, index) => `
                <div class="alert alert-light border qa-card">
                    <strong>Q:</strong> ${qa.Ques}<br>
                    <div class="correctAns"> <strong>A:</strong> ${qa.Ans} </div>
                    <div class="mt-2 quizScreen">
                        <textarea id="textArea${index}" class="form-control"></textarea>
                        <div class="row mt-2">
                            <button class="btn btn-sm btn-outline-dark col-md-4" onclick="speakQues('${index}')">
                                <i class="bi bi-volume-up"></i> Play Question
                            </button>
                            <button id="listenBtn${index}" class="btn btn-sm btn-outline-dark listen-button col-md-4" onclick="toggleStartStop(this, '${index}')">
                                <i class="bi bi-mic"></i> Start Listening
                            </button>
                            <button id="feedbackBtn${index}" class="btn btn-sm btn-outline-dark col-md-4" data-bs-toggle="modal" data-bs-target="#myModal${index}" onclick="showFeedback('${index}')">
                                <i class="bi bi-clipboard-check"></i> Check response
                            </button>
                        </div>

                        <div class="modal fade" id="myModal${index}" tabindex="-1" aria-labelledby="modalLabel${index}" aria-hidden="true">
                            <div class="modal-dialog modal-lg">
                                <div class="modal-content">
                                    <div class="modal-header">
                                        <h5 class="modal-title" id="modalLabel${index}">Feedback for question ${index+1}</h5>
                                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                                    </div>
                                    <div class="modal-body" id="modalBody${index}">
                                    </div>
                                    <div class="modal-footer">
                                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                                    </div>
                                </div>
                            </div>
                        </div>
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
        uploadBtn_study.disabled = false;
        uploadBtn_quiz.disabled = false;
        loadingSpinner.classList.add("d-none");
        responseScreen.classList.remove("d-none");
        responseScreen.classList.add("d-flex");
        if(quizMode) {
            document.getElementById('cleanedImageDiv').classList.add("d-none");
            const elements = document.getElementsByClassName('correctAns');
            while(elements.length > 0){
                elements[0].parentNode.removeChild(elements[0]);
            }
            document.getElementById('QADiv').classList.remove('col-md-6');
            document.getElementById('QADiv').classList.add('col-md-12');
        } else {
            const elements = document.getElementsByClassName('quizScreen');
            while(elements.length > 0){
                elements[0].parentNode.removeChild(elements[0]);
            }
            document.getElementById('uploadFeedback_btn').classList.add("d-none");
        }
    });
}

function showFeedback(index) {
    speakIndex = index;
    const textArea = document.getElementById('textArea'+speakIndex);
    checkAnswer(textArea.value, correctAnswer[speakIndex], questions[speakIndex]);
}

function speakQues(index) {
    var questionText = questions[index];
    const speechSynth = window.speechSynthesis;
    if (!speechSynth.speaking && questionText.trim().length) {
        const newUtter = new SpeechSynthesisUtterance(questionText);
        speechSynth.speak(newUtter);
    }
}

function checkAnswer(userAnswer, correctAnswer, question) {
    var modalBody = document.getElementById('modalBody'+speakIndex);
    if(userAnswer==undefined || userAnswer=="")
        modalBody.innerHTML = "<div class='text-dark text-center'></div><p>Error</p>";
    else {
        modalBody.innerHTML = "<div class='spinner-border text-dark text-center' role='status'></div><p>Processing...</p>";
        var answers = JSON.stringify({'userAnswer': userAnswer,'correctAnswer': correctAnswer, 'question': question});
        fetch('https://'+projectId+'.uc.r.appspot.com/checkAnswer', {
            method: 'POST',
            headers: {
                'Accept': 'application/json, text/plain',
                'Content-Type': 'application/json;charset=UTF-8'
            },
            body: answers
        })
        .then(response => response.json())  
        .then(data => {
            feedback[speakIndex] = data;
            const feedbackBtn = document.getElementById('feedbackBtn'+speakIndex);
            feedbackBtn.innerHTML = 'Accuracy: '+feedback[speakIndex].data.Overall_Score;
            modalBody.innerHTML = "<pre> "+JSON.stringify(feedback[speakIndex]['data'], null, 2)+"</pre>";
        });
    }
}

function downloadPdf() {
    pdfText = pdfResponse.map((obj, index) => 'Ques'+ (index + 1) + ': ' + obj.Ques + '\nAns: ' + obj.Ans).join("\n\n");
    var doc = new jsPDF({
        orientation: 'p',
        unit: 'mm',
        format: 'a4',
        putOnlyUsedFonts:true
       });
    const marginLeft = 10;
    const marginTop = 20;
    const pageWidth = 190;
    const lines = doc.splitTextToSize(pdfText, pageWidth);
    let y = marginTop;
    lines.forEach((line, index) => {
        if (y > 280) {
            doc.addPage();
            y = marginTop;
        }
        doc.text(line, marginLeft, y);
        y += 10;
    });
    doc.save("Generated QA.pdf");
}

function uploadFeedback() {
    fetch('https://'+projectId+'.uc.r.appspot.com/uploadFeedback', {
        method: 'POST',
        headers: {
            'Accept': 'application/json, text/plain',
            'Content-Type': 'application/json;charset=UTF-8'
        },
        body: JSON.stringify({'user': document.getElementById('userNameInput').value,'feedback': feedback})
    })
    .then(response => response.json())  
    .then(data => {
        console.log(data);
    });
};

function downloadCleanImage() {
    fetch(cleanedImageUrl, {
        mode : 'no-cors'
    })
        .then(response => response.blob())
        .then(blob => {
            const link = document.createElement("a");
            link.target="_blank";
            link.href = cleanedImageUrl;
            link.download = "cleaned_image.png"; // Make sure to set the filename
            document.body.appendChild(link);
            link.click();
            link.remove();
        })
        .catch(error => console.error("Download error:", error));
}