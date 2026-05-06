// Dashboard interactions:
//   - sign-out button (POST /api/auth/logout, redirect to login)
//   - hydrate the build-time UTC timestamp into viewer-local time
(function () {
  var btn = document.getElementById('rp-signout');
  if (btn) {
    btn.addEventListener('click', function () {
      btn.disabled = true;
      btn.textContent = 'Signing out…';
      // keepalive lets the cookie-clear POST finish after we navigate,
      // so we can redirect immediately instead of waiting on the
      // function's cold start.
      try {
        fetch('/api/auth/logout', {
          method: 'POST',
          credentials: 'same-origin',
          keepalive: true,
        }).catch(function () {});
      } catch (e) {}
      requestAnimationFrame(function () {
        var top = window.top || window.parent || window;
        try { top.location.replace('/login.html'); }
        catch (e) { window.location.replace('/login.html'); }
      });
    });
  }

  // Hydrate the last-updated timestamp into the viewer's local time.
  // The element's datetime attribute is the build-time UTC ISO string.
  // Force "DD MMM YYYY at HH:MM" (24h) regardless of viewer locale, to
  // match the date format used everywhere else in the dashboard.
  var t = document.getElementById('rp-last-updated');
  if (t) {
    var iso = t.getAttribute('datetime');
    var d = iso ? new Date(iso) : null;
    if (d && !isNaN(d.getTime())) {
      var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
      var pad = function (n) { return n < 10 ? '0' + n : '' + n; };
      t.textContent = d.getDate() + ' ' + months[d.getMonth()] + ' ' + d.getFullYear()
        + ' at ' + pad(d.getHours()) + ':' + pad(d.getMinutes());
    }
  }
})();
