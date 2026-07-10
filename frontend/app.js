const FIELD_HELP = {
  anaemia: 'Reduced red blood cells or haemoglobin. Select Yes if the patient has anaemia.',
  ejection_fraction:
    'Percentage of blood leaving the heart with each beat. Normal is often around 50 to 70 percent. Lower values can indicate weaker heart function.',
  creatinine_phosphokinase:
    'CPK enzyme level, often measured in mcg/L. Higher values may reflect muscle or heart stress.',
  serum_creatinine:
    'Kidney function marker measured in mg/dL. Higher values can indicate reduced kidney performance.',
  serum_sodium:
    'Blood sodium level in mEq/L. Typical adult range is often around 135 to 145.',
  time: 'Number of days during follow-up before the outcome was recorded.',
}

const form = document.getElementById('predict-form')
const submitBtn = document.getElementById('submit-btn')
const resetBtn = document.getElementById('reset-btn')
const errorBox = document.getElementById('error')
const resultCard = document.getElementById('result-card')
const riskBadge = document.getElementById('risk-badge')
const probabilityEl = document.getElementById('probability')
const messageEl = document.getElementById('message')
const whyBlock = document.getElementById('why-block')
const whyList = document.getElementById('why-list')
const riskMeter = document.getElementById('risk-meter')
const riskMeterFill = document.getElementById('risk-meter-fill')

function initHelpPopovers() {
  for (const tip of document.querySelectorAll('.info-tip[data-help]')) {
    const key = tip.dataset.help
    const popover = tip.querySelector('.info-popover')
    if (!popover || !FIELD_HELP[key]) continue
    popover.textContent = FIELD_HELP[key]
  }
}

function resetResults() {
  resultCard.classList.remove('show')
  errorBox.classList.remove('show')
  whyBlock.hidden = true
  whyList.innerHTML = ''
}

form.addEventListener('submit', async (event) => {
  event.preventDefault()
  errorBox.classList.remove('show')
  submitBtn.disabled = true
  submitBtn.textContent = 'Running...'

  const formData = new FormData(form)
  const payload = Object.fromEntries(formData.entries())

  for (const key of Object.keys(payload)) {
    if (['age', 'ejection_fraction', 'serum_creatinine', 'time'].includes(key)) {
      payload[key] = Number.parseFloat(payload[key])
    } else {
      payload[key] = Number.parseInt(payload[key], 10)
    }
  }

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    const data = await response.json()
    if (!response.ok) {
      throw new Error(data.detail || 'Assessment failed')
    }

    const isHigh = data.risk === 'high'
    riskBadge.textContent = isHigh ? 'Higher risk' : 'Lower risk'
    riskBadge.className = `risk-badge ${isHigh ? 'high' : 'low'}`
    probabilityEl.textContent = `${data.probability_percent}%`
    messageEl.textContent = data.message
    riskMeter.className = `meter ${isHigh ? 'high' : ''}`
    riskMeterFill.style.width = `${Math.min(100, Math.max(0, data.probability_percent))}%`

    whyList.innerHTML = ''
    if (Array.isArray(data.why) && data.why.length > 0) {
      for (const factor of data.why) {
        const item = document.createElement('li')
        item.textContent = factor.summary
        whyList.appendChild(item)
      }
      whyBlock.hidden = false
    } else {
      whyBlock.hidden = true
    }

    resultCard.classList.add('show')
  } catch (error) {
    errorBox.textContent = error instanceof Error ? error.message : 'Assessment failed'
    errorBox.classList.add('show')
  } finally {
    submitBtn.disabled = false
    submitBtn.textContent = 'Run assessment'
  }
})

resetBtn.addEventListener('click', () => {
  window.setTimeout(resetResults, 0)
})

initHelpPopovers()
