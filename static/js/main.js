document.addEventListener('DOMContentLoaded', () => {
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href === '#') return;
            
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                const offsetTop = target.offsetTop - 80; // Account for fixed navbar
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
                
                // Close mobile menu if open
                const mobileMenu = document.getElementById('mobileMenu');
                if (mobileMenu) {
                    mobileMenu.style.display = 'none';
                }
            }
        });
    });

    // Hamburger Menu Toggle
    const hamburger = document.querySelector('.hamburger');
    const mobileMenu = document.getElementById('mobileMenu');

    if (hamburger && mobileMenu) {
        hamburger.addEventListener('click', () => {
            const isHidden = mobileMenu.style.display === 'none' || mobileMenu.style.display === '';
            mobileMenu.style.display = isHidden ? 'block' : 'none';
        });

        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!hamburger.contains(e.target) && !mobileMenu.contains(e.target)) {
                mobileMenu.style.display = 'none';
            }
        });
    }

    // Navbar background on scroll
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        window.addEventListener('scroll', () => {
            if (window.scrollY > 50) {
                navbar.style.background = 'rgba(255, 255, 255, 0.95)';
                navbar.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.1)';
            } else {
                navbar.style.background = 'rgba(255, 255, 255, 0.8)';
                navbar.style.boxShadow = 'none';
            }
        });
    }

    // Prediction Form Handling
    const form = document.getElementById('predictionForm');
    const resultContainer = document.getElementById('resultContainer');

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const submitBtn = form.querySelector('button[type="submit"]');
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            submitBtn.disabled = true;

            const formData = new FormData(form);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    resultContainer.style.display = 'block';

                    let html = '';
                    if (data.prediction === 1) {
                        // High Risk Result
                        html = `
                            <div style="
                                background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.05));
                                border: 2px solid rgba(239, 68, 68, 0.3);
                                border-radius: 1.5rem;
                                padding: 3rem;
                                text-align: center;
                                animation: fadeInUp 0.6s ease-out;
                            ">
                                <div style="
                                    width: 5rem;
                                    height: 5rem;
                                    background: linear-gradient(135deg, #ef4444, #dc2626);
                                    border-radius: 50%;
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    margin: 0 auto 1.5rem;
                                    box-shadow: 0 10px 25px rgba(239, 68, 68, 0.3);
                                ">
                                    <i class="fas fa-exclamation-triangle" style="font-size: 2.5rem; color: white;"></i>
                                </div>
                                <h3 style="
                                    font-size: 2rem;
                                    font-weight: 700;
                                    color: #991b1b;
                                    margin-bottom: 1rem;
                                    font-family: 'Poppins', sans-serif;
                                ">Higher Risk Detected</h3>
                                <p style="
                                    font-size: 1.125rem;
                                    color: #7f1d1d;
                                    margin-bottom: 1.5rem;
                                    line-height: 1.7;
                                ">Our AI model indicates a significant probability of cardiovascular disease.</p>
                                ${data.probability ? `
                                    <div style="
                                        background: rgba(255, 255, 255, 0.8);
                                        border-radius: 1rem;
                                        padding: 1.5rem;
                                        margin-bottom: 1.5rem;
                                        display: inline-block;
                                    ">
                                        <div style="
                                            font-size: 0.875rem;
                                            text-transform: uppercase;
                                            letter-spacing: 0.1em;
                                            color: #64748b;
                                            margin-bottom: 0.5rem;
                                            font-weight: 600;
                                        ">Risk Probability</div>
                                        <div style="
                                            font-size: 3rem;
                                            font-weight: 800;
                                            background: linear-gradient(135deg, #ef4444, #dc2626);
                                            -webkit-background-clip: text;
                                            -webkit-text-fill-color: transparent;
                                            background-clip: text;
                                            font-family: 'Poppins', sans-serif;
                                        ">${data.probability}%</div>
                                    </div>
                                ` : ''}
                                <div style="
                                    background: rgba(255, 255, 255, 0.9);
                                    border-radius: 1rem;
                                    padding: 1.5rem;
                                    margin-top: 2rem;
                                    border-left: 4px solid #ef4444;
                                ">
                                    <p style="
                                        color: #374151;
                                        margin: 0;
                                        font-size: 0.95rem;
                                        line-height: 1.7;
                                    ">
                                        <strong style="color: #991b1b;">Important:</strong> Please consult a qualified healthcare professional immediately for a thorough examination and personalized medical advice. This assessment is for informational purposes only.
                                    </p>
                                </div>
                            </div>
                        `;
                    } else {
                        // Low Risk Result
                        html = `
                            <div style="
                                background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.05));
                                border: 2px solid rgba(16, 185, 129, 0.3);
                                border-radius: 1.5rem;
                                padding: 3rem;
                                text-align: center;
                                animation: fadeInUp 0.6s ease-out;
                            ">
                                <div style="
                                    width: 5rem;
                                    height: 5rem;
                                    background: linear-gradient(135deg, #10b981, #059669);
                                    border-radius: 50%;
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    margin: 0 auto 1.5rem;
                                    box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
                                ">
                                    <i class="fas fa-check-circle" style="font-size: 2.5rem; color: white;"></i>
                                </div>
                                <h3 style="
                                    font-size: 2rem;
                                    font-weight: 700;
                                    color: #065f46;
                                    margin-bottom: 1rem;
                                    font-family: 'Poppins', sans-serif;
                                ">Low Risk Detected</h3>
                                <p style="
                                    font-size: 1.125rem;
                                    color: #047857;
                                    margin-bottom: 1.5rem;
                                    line-height: 1.7;
                                ">Our AI model indicates a low probability of cardiovascular disease based on the provided information.</p>
                                ${data.probability ? `
                                    <div style="
                                        background: rgba(255, 255, 255, 0.8);
                                        border-radius: 1rem;
                                        padding: 1.5rem;
                                        margin-bottom: 1.5rem;
                                        display: inline-block;
                                    ">
                                        <div style="
                                            font-size: 0.875rem;
                                            text-transform: uppercase;
                                            letter-spacing: 0.1em;
                                            color: #64748b;
                                            margin-bottom: 0.5rem;
                                            font-weight: 600;
                                        ">Risk Probability</div>
                                        <div style="
                                            font-size: 3rem;
                                            font-weight: 800;
                                            background: linear-gradient(135deg, #10b981, #059669);
                                            -webkit-background-clip: text;
                                            -webkit-text-fill-color: transparent;
                                            background-clip: text;
                                            font-family: 'Poppins', sans-serif;
                                        ">${data.probability}%</div>
                                    </div>
                                ` : ''}
                                <div style="
                                    background: rgba(255, 255, 255, 0.9);
                                    border-radius: 1rem;
                                    padding: 1.5rem;
                                    margin-top: 2rem;
                                    border-left: 4px solid #10b981;
                                ">
                                    <p style="
                                        color: #374151;
                                        margin: 0;
                                        font-size: 0.95rem;
                                        line-height: 1.7;
                                    ">
                                        <strong style="color: #065f46;">Great news!</strong> Continue maintaining a healthy lifestyle with regular exercise, balanced nutrition, and routine health check-ups. Remember to consult healthcare professionals for personalized medical advice.
                                    </p>
                                </div>
                            </div>
                        `;
                    }
                    resultContainer.innerHTML = html;

                    // Scroll to result with smooth behavior
                    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

                } else {
                    alert('Error: ' + (data.error || 'An error occurred while processing your request.'));
                }

            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred while processing your request. Please try again.');
            } finally {
                submitBtn.innerHTML = originalBtnText;
                submitBtn.disabled = false;
            }
        });
    }

    // Add fade-in animation for results
    const style = document.createElement('style');
    style.textContent = `
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    `;
    document.head.appendChild(style);
});