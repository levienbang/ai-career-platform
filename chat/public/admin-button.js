// Adds a "Quản trị" button next to the send button. One click sends "admin",
// which the app answers with the admin toolbar (stats, skill review, index...).
// Chainlit's own command buttons only pre-select a mode and still need a send.
(function () {
  const BUTTON_ID = "admin-tools-button";

  function sendAdmin() {
    const input = document.getElementById("chat-input");
    if (!input) return;
    // Set the textarea through the native setter so React sees the change.
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set;
    setter.call(input, "admin");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    // React enables the send button on its next render; wait for it, then click.
    let tries = 0;
    const timer = setInterval(() => {
      const submit = document.getElementById("chat-submit");
      if (submit && !submit.disabled) {
        clearInterval(timer);
        submit.click();
      } else if (++tries > 40) {
        clearInterval(timer);
      }
    }, 25);
  }

  function addButton() {
    const submit = document.getElementById("chat-submit");
    if (!submit || document.getElementById(BUTTON_ID)) return;
    const button = document.createElement("button");
    button.id = BUTTON_ID;
    button.type = "button";
    button.title = "Công cụ quản trị";
    button.textContent = "⚙️ Quản trị";
    button.style.cssText =
      "margin-right:8px;padding:4px 12px;border-radius:9999px;font-size:13px;" +
      "border:1px solid rgba(127,127,127,.4);background:transparent;color:inherit;cursor:pointer";
    button.addEventListener("click", sendAdmin);
    submit.parentElement.insertBefore(button, submit);
  }

  new MutationObserver(addButton).observe(document.body, { childList: true, subtree: true });
})();
