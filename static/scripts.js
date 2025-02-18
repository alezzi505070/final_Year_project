/* scripts.js
   Enhanced for Brown/Gold Lux Theme,
   with parallax, tilt, custom pointer, etc.
   in full detail
*/

document.addEventListener("DOMContentLoaded", () => {
  const analyzeForm = document.getElementById("analyzeForm");
  const loadingOverlay = document.getElementById("loadingOverlay");
  const clearIcon = document.querySelector(".clear-icon");
  const statusMessage = document.getElementById("statusMessage");
  const symbolInput = document.querySelector(".symbol-input");

  // Clear input on click of 'x' icon
  if (clearIcon && symbolInput) {
    clearIcon.addEventListener("click", () => {
      symbolInput.value = "";
      symbolInput.focus();
    });
  }

  // On form submit, show overlay, then fetch
  if (analyzeForm && loadingOverlay) {
    analyzeForm.addEventListener("submit", function (e) {
      e.preventDefault();
      const stockSymbol = symbolInput.value.trim();

      // If empty, show error
      if (!stockSymbol) {
        statusMessage.textContent = "Please enter a stock symbol.";
        statusMessage.classList.add("show");
        setTimeout(() => {
          statusMessage.classList.remove("show");
        }, 3000);
        return;
      }

      // Show loading overlay
      loadingOverlay.classList.add("show");

      // Use fetch to send POST
      fetch(analyzeForm.action, {
        method: "POST",
        body: new FormData(analyzeForm),
      })
        .then((response) => response.text())
        .then((data) => {
          // Replace entire body
          document.body.innerHTML = data;
          window.scrollTo({ top: 0 });
        })
        .catch((error) => {
          console.error("There was an error:", error);
          loadingOverlay.classList.remove("show");
          statusMessage.textContent = "An error occurred. Please try again.";
          statusMessage.classList.add("show");
          setTimeout(() => {
            statusMessage.classList.remove("show");
          }, 3000);
        });
    });
  }

  // If there's a back-link, handle smooth scroll up, then redirect
  const backLink = document.querySelector(".back-link");
  if (backLink) {
    backLink.addEventListener("click", (e) => {
      e.preventDefault();
      window.scrollTo({ top: 0, behavior: "smooth" });
      setTimeout(() => {
        window.location.href = backLink.href;
      }, 500);
    });
  }
});
