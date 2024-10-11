import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Configure the page layout
st.set_page_config(layout='wide')

def generate_data(num_points: int = 100, true_a: float = 2.0, true_b: float = 1.0, noise_std: float = 2.0) -> tuple:
    """
    Generates noisy data points along a line Y = true_a * X + true_b + noise.

    Args:
        num_points (int): Number of data points to generate.
        true_a (float): True slope of the underlying line.
        true_b (float): True intercept of the underlying line.
        noise_std (float): Standard deviation of the Gaussian noise added to Y.

    Returns:
        tuple: Arrays of X and Y data points.
    """
    X = np.linspace(-10, 10, num_points)
    noise = np.random.normal(0, noise_std, num_points)
    Y = true_a * X + true_b + noise
    return X, Y

def compute_loss(a: float, b: float, X: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes the mean squared error loss for the line Y = a * X + b.

    Args:
        a (float): Slope of the line.
        b (float): Intercept of the line.
        X (np.ndarray): Array of X data points.
        Y (np.ndarray): Array of Y data points.

    Returns:
        float: Computed mean squared error loss.
    """
    N = len(X)
    Y_pred = a * X + b
    loss = (1 / N) * np.sum((Y - Y_pred) ** 2)
    return loss

def compute_gradients(a: float, b: float, X: np.ndarray, Y: np.ndarray) -> tuple:
    """
    Computes the gradients of the loss function with respect to a and b.

    Args:
        a (float): Current value of slope.
        b (float): Current value of intercept.
        X (np.ndarray): Array of X data points.
        Y (np.ndarray): Array of Y data points.

    Returns:
        tuple: Gradients with respect to a and b.
    """
    N = len(X)
    Y_pred = a * X + b
    dL_da = (-2 / N) * np.sum(X * (Y - Y_pred))
    dL_db = (-2 / N) * np.sum(Y - Y_pred)
    return dL_da, dL_db

# Create columns for layout
control_col, plot_col1, plot_col2 = st.columns([1, 2, 2])

with control_col:
    st.header("Controls")
    # Learning rate slider
    learning_rate = st.slider('Learning Rate', min_value=0.00001, max_value=0.1,
                              value=0.001, step=0.00001, format="%.5f")
    # Buttons
    reset_button = st.button('Reset')
    step_button = st.button('Step')

# Initialize or reset the session state
if 'initialized' not in st.session_state or reset_button:
    # Generate data points
    X_data, Y_data = generate_data()
    st.session_state.X_data = X_data
    st.session_state.Y_data = Y_data
    # Initialize parameters a and b randomly
    a_init = np.random.uniform(-5, 5)
    b_init = np.random.uniform(-5, 5)
    st.session_state.a = a_init
    st.session_state.b = b_init
    # Initialize history lists for plotting the optimization path
    st.session_state.a_history = [a_init]
    st.session_state.b_history = [b_init]
    st.session_state.loss_history = [compute_loss(a_init, b_init, X_data, Y_data)]
    st.session_state.initialized = True
    # Store fixed axis limits for the 2D plot
    st.session_state.xmin = -12
    st.session_state.xmax = 12
    st.session_state.ymin = min(Y_data) - 5
    st.session_state.ymax = max(Y_data) + 5

# Perform one step of gradient descent when 'Step' button is clicked
if step_button:
    X_data = st.session_state.X_data
    Y_data = st.session_state.Y_data
    a = st.session_state.a
    b = st.session_state.b
    # Compute gradients
    dL_da, dL_db = compute_gradients(a, b, X_data, Y_data)
    # Update parameters
    a_new = a - learning_rate * dL_da
    b_new = b - learning_rate * dL_db
    st.session_state.a = a_new
    st.session_state.b = b_new
    # Update history
    st.session_state.a_history.append(a_new)
    st.session_state.b_history.append(b_new)
    st.session_state.loss_history.append(compute_loss(a_new, b_new, X_data, Y_data))

with plot_col1:
    st.header("Data and Fitted Line")
    # Plot the data points and the current fitted line
    fig, ax = plt.subplots()
    X_data = st.session_state.X_data
    Y_data = st.session_state.Y_data
    a = st.session_state.a
    b = st.session_state.b
    ax.scatter(X_data, Y_data, label='Data Points', color='blue')
    X_line = np.linspace(st.session_state.xmin, st.session_state.xmax, 100)
    Y_line = a * X_line + b
    ax.plot(X_line, Y_line, color='red', label='Fitted Line')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    # Fix the axis limits
    ax.set_xlim(st.session_state.xmin, st.session_state.xmax)
    ax.set_ylim(st.session_state.ymin, st.session_state.ymax)
    st.pyplot(fig)

with plot_col2:
    st.header("Loss Surface and Optimization Path")
    # Determine the range for a and b based on initial settings
    # Fixed range for better visualization of the path
    a_range = np.linspace(-10, 10, 50)
    b_range = np.linspace(-10, 10, 50)
    A, B = np.meshgrid(a_range, b_range)
    # Compute the loss over the grid
    Loss = np.zeros_like(A)
    X_data = st.session_state.X_data
    Y_data = st.session_state.Y_data
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            Loss[i, j] = compute_loss(A[i, j], B[i, j], X_data, Y_data)
    # Create the 3D surface plot
    fig = go.Figure()
    fig.add_trace(go.Surface(x=A, y=B, z=Loss, colorscale='Viridis', opacity=0.7, showscale=False))
    # Add the optimization path (excluding the last point)
    a_history = st.session_state.a_history
    b_history = st.session_state.b_history
    loss_history = st.session_state.loss_history
    if len(a_history) > 1:
        fig.add_trace(go.Scatter3d(
            x=a_history[:-1],
            y=b_history[:-1],
            z=loss_history[:-1],
            mode='lines+markers',
            marker=dict(size=4, color='blue'),
            line=dict(color='blue', width=2),
            showlegend=False
        ))
    # Add the current point
    fig.add_trace(go.Scatter3d(
        x=[a_history[-1]],
        y=[b_history[-1]],
        z=[loss_history[-1]],
        mode='markers',
        marker=dict(size=6, color='red'),
        showlegend=False
    ))
    # Update plot layout
    fig.update_layout(
        scene=dict(
            xaxis_title='a (Slope)',
            yaxis_title='b (Intercept)',
            zaxis_title='Loss',
            xaxis=dict(range=[-10, 10]),
            yaxis=dict(range=[-10, 10]),
            zaxis=dict(range=[np.min(Loss), np.max(Loss)])
        ),
        width=700,
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False  # Remove the legend
    )
    st.plotly_chart(fig)
