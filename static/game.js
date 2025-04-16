const images = [
  '/static/assets/hello.jpg', '/static/assets/hello1.jpg',
  '/static/assets/goodbye.jpg', '/static/assets/goodbye1.jpg',
  '/static/assets/please.jpg', '/static/assets/please1.jpg',
  '/static/assets/yes.jpg', '/static/assets/yes1.jpg',
  '/static/assets/no.jpg', '/static/assets/no1.jpg',
  '/static/assets/Thanks.jpg', '/static/assets/Thanks1.jpg',
];

let attempts = 0;
let firstCard = null;
let secondCard = null;
let lockBoard = false;

function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

function createBoard() {
  const gameBoard = document.getElementById('game-board');
  gameBoard.innerHTML = ''; // Clear previous board
  shuffle(images);
  
  images.forEach(image => {
      const card = document.createElement('div');
      card.classList.add('card');
      card.dataset.image = image;
      
      // Front and Back elements
      const front = document.createElement('div');
      front.classList.add('front');
      front.style.background = "#ccc"; // Placeholder for front

      const back = document.createElement('div');
      back.classList.add('back');
      const img = document.createElement('img');
      img.src = image;
      back.appendChild(img);

      card.appendChild(front);
      card.appendChild(back);

      card.addEventListener('click', flipCard);
      gameBoard.appendChild(card);
  });
}


function flipCard() {
    if (lockBoard || this === firstCard) return;
    
    this.classList.add('flipped');

    if (!firstCard) {
        firstCard = this;
        return;
    }

    secondCard = this;
    attempts++;
    document.getElementById('attempts').textContent = attempts;

    checkForMatch();
}

function checkForMatch() {
    lockBoard = true;
    
    const firstImageBase = firstCard.dataset.image.replace(/\d+\.jpg$/, '.jpg');
    const secondImageBase = secondCard.dataset.image.replace(/\d+\.jpg$/, '.jpg');
    
    if (firstImageBase === secondImageBase) {
        firstCard.classList.add('matched');
        secondCard.classList.add('matched');
        firstCard.removeEventListener('click', flipCard);
        secondCard.removeEventListener('click', flipCard);
        resetBoard();
    } else {
        setTimeout(() => {
            firstCard.classList.remove('flipped');
            secondCard.classList.remove('flipped');
            resetBoard();
        }, 1000);
    }
}

function resetBoard() {
    [firstCard, secondCard] = [null, null];
    lockBoard = false;
}

// Start the game
createBoard();
