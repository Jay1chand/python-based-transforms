from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import mpld3
import sympy as sp

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('homepage.html')

########## *** Fourier Transform Code *** ###########
default_function = "sin(2*pi*x) + sin(pi*x) + sin(5*pi*x) + sin(2.5*pi*x) + sin(3.33*pi*x)"

@app.route('/fourier', methods=['GET', 'POST'])
def fourier():
    if request.method == 'POST':
        func_str = request.form['function']
    else:
        func_str = default_function
    
    # Generate x values
    N = 1000            # Number of points
    L = 150.0           # Length of interval
    dx = L / N          # Sampling interval
    x = np.linspace(-L/2, L/2, N, endpoint=False)  # x values

    # Convert string function to a lambda function
    x_sym = sp.symbols('x')
    try:
        func = sp.lambdify(x_sym, sp.sympify(func_str), 'numpy')
        y = func(x)
    except Exception as e:
        return render_template('fourier.html', plot_html="", error=f"Error in function: {e}")

    # Compute Fourier Transform manually
    def manual_dft(y):
        N = len(y)
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, y)

    fourier = manual_dft(y)
    
    # Shift the zero-frequency component to the center
    fourier = np.fft.fftshift(fourier)

    # Frequencies for plotting the Fourier transform
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    # Create the plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    axs[0].plot(x, y, label='Original Function')
    axs[0].set_title('Original Function')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()

    axs[1].plot(freq, np.abs(fourier), label='Fourier Transform')
    axs[1].set_title('Fourier Transform of Function')
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Amplitude')
    axs[1].legend()

    plt.tight_layout()

    # Convert plot to HTML
    plot_html = mpld3.fig_to_html(fig)
    return render_template('fourier.html', plot_html=plot_html, error=None)

########## *** Fourier Transform Code Ends *** ###########

############# *** Laplace Transform Code *** #############
def laplace_transform_num(f, s, upper_limit=np.inf):
    integrand = lambda t: np.exp(-s * t) * f(t)
    result, _ = quad(integrand, 0, upper_limit)
    return result

@app.route('/laplace', methods=['GET', 'POST'])
def laplace():
    return render_template('laplace.html')

@app.route('/update_plot', methods=['POST'])
def update_plot():
    # Get function expression from POST request
    data = request.json
    function_expression = data['functionExpression']

    # Define symbols and create the function f(t)
    t = sp.symbols('t')
    heaviside = sp.Heaviside(t)
    f_sym = sp.lambdify(t, function_expression, modules=['numpy', {'Heaviside': heaviside}])

    # Generate values for t and s and compute f(t) and F(s) for these values
    t_vals = np.linspace(0, 100, 400)
    f_vals = f_sym(t_vals)
    
    s_vals = np.linspace(0.1, 100, 400)
    F_s_vals = [laplace_transform_num(f_sym, s) for s in s_vals]

    # Prepare data to send back as JSON
    response_data = {
        't_vals': t_vals.tolist(),
        'f_vals': f_vals.tolist(),
        's_vals': s_vals.tolist(),
        'F_s_vals_real': [result.real for result in F_s_vals],
        'F_s_vals_imag': [result.imag for result in F_s_vals]
    }

    return jsonify(response_data)

#### ***Z TRANSFORM CODE*** ###

# Z-transform function
def z_transform_z_func(expr, n_expr, max_terms=100):
    z, m = sp.symbols('z m')
    return sp.Sum(expr.subs(n_expr, m) * z**(-m), (m, 0, max_terms)).doit()

@app.route('/ztransform', methods=['GET', 'POST'])
def ztransform():
    if request.method == 'POST':
        try:
            # Get the function input from the form
            str_exp = request.form['function']
            expr = sp.sympify(str_exp)  # Convert string input to symbolic expression

            # Check if the input is a constant (like '1')
            if expr.is_constant():
                z_transform = 1 / (1 - sp.symbols('z'))  # For constant input, Z-transform is 1/(1 - z)
            else:
                # Compute the Z-transform (using a finite number of terms for numerical stability)
                z_transform = z_transform_z_func(expr, sp.symbols('g'), max_terms=100)

            # Return the computed Z-transform as a string
            return render_template('ztransform.html', z_transform=str(z_transform), error=None)

        except Exception as e:
            # Handle any errors (e.g., invalid input or symbolic evaluation issues)
            error_message = f"An error occurred: {str(e)}"
            return render_template('ztransform.html', z_transform=None, error=error_message)
    
    # On GET request, render an empty form
    return render_template('ztransform.html', z_transform=None, error=None)


########## *** Z-Transform Code Ends *** ###########
########## *** INVERSE Z-Transform Code *** ########










######### **** INVERSE Z TRANSFORM CODE ENDS **** #########

if __name__ == '__main__':
    app.run(debug=True)
