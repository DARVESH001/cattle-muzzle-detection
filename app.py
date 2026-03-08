import streamlit as st
from PIL import Image
from model.inference import MuzzleIdentifier

st.set_page_config(
    page_title="Cattle Muzzle ID System",
    page_icon="🐄",
    layout="wide",
)


@st.cache_resource
def load_identifier():
    return MuzzleIdentifier()


def main():
    st.title("Cattle Muzzle Identification System")

    identifier = load_identifier()

    if not identifier.is_ready():
        st.error(
            "Model not loaded! Train the model first by running:\n\n"
            "```\ncd cattle_muzzle_detection\npython model/train.py\n```"
        )
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "Check Image",
        "Compare Two Images",
        "Register Cattle",
        "Registry",
    ])

    # --- Tab 1: Check Image ---
    with tab1:
        st.header("Check Cattle Muzzle Image")
        st.write("Upload an image to check if it's a cattle muzzle and find matches in the registry.")

        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="check")

        if uploaded:
            image = Image.open(uploaded).convert('RGB')
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)

            with col2:
                with st.spinner("Analyzing image..."):
                    # Step 1: Is it a muzzle?
                    muzzle_result = identifier.is_cattle_muzzle(image)

                if muzzle_result["is_muzzle"]:
                    st.success(
                        f"This IS a cattle muzzle image "
                        f"(Confidence: {muzzle_result['confidence']:.2%} - {muzzle_result['confidence_label']})"
                    )

                    # Step 2: Search registry
                    matches = identifier.identify_cattle(image)
                    if matches:
                        st.subheader("Registry Search Results")
                        top_match = matches[0]

                        if top_match["is_match"]:
                            st.success(
                                f"Match found! This cattle is most likely **{top_match['name']}** "
                                f"(Similarity: {top_match['similarity']:.2%})"
                            )
                        else:
                            st.warning(
                                "No confident match found. This may be a new cattle not in the registry."
                            )

                        # Show all matches
                        for m in matches:
                            color = "🟢" if m["is_match"] else "🔴"
                            st.write(
                                f"{color} **{m['name']}** — "
                                f"Similarity: {m['similarity']:.2%} "
                                f"({m['confidence_label']})"
                            )
                            st.progress(max(0.0, min(1.0, m['similarity'])))
                    else:
                        st.info("No cattle registered yet. Register some cattle first!")
                else:
                    st.error(
                        f"This does NOT appear to be a cattle muzzle image "
                        f"(Confidence: {muzzle_result['confidence']:.2%} - {muzzle_result['confidence_label']})"
                    )

    # --- Tab 2: Compare Two Images ---
    with tab2:
        st.header("Compare Two Muzzle Images")
        st.write("Upload two images to check if they belong to the same cattle.")

        col1, col2 = st.columns(2)
        with col1:
            img1_file = st.file_uploader("Image 1", type=["jpg", "jpeg", "png"], key="cmp1")
        with col2:
            img2_file = st.file_uploader("Image 2", type=["jpg", "jpeg", "png"], key="cmp2")

        if img1_file and img2_file:
            img1 = Image.open(img1_file).convert('RGB')
            img2 = Image.open(img2_file).convert('RGB')

            col1, col2 = st.columns(2)
            with col1:
                st.image(img1, caption="Image 1", use_column_width=True)
            with col2:
                st.image(img2, caption="Image 2", use_column_width=True)

            if st.button("Compare", key="compare_btn"):
                with st.spinner("Comparing images..."):
                    result = identifier.compare_images(img1, img2)

                st.divider()

                if result["same_cattle"]:
                    st.success(f"SAME CATTLE — Similarity: {result['similarity']:.2%}")
                else:
                    st.error(f"DIFFERENT CATTLE — Similarity: {result['similarity']:.2%}")

                st.metric("Similarity Score", f"{result['similarity']:.2%}")
                st.write(f"Confidence: **{result['confidence_label']}**")
                st.progress(max(0.0, min(1.0, result['similarity'])))

    # --- Tab 3: Register Cattle ---
    with tab3:
        st.header("Register New Cattle")
        st.write("Upload multiple muzzle images of the same cattle to register it in the system.")

        cattle_name = st.text_input("Cattle Name / ID", placeholder="e.g., Cattle_001 or Lakshmi")

        images_files = st.file_uploader(
            "Upload muzzle images (at least 3 recommended)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="register",
        )

        if images_files:
            cols = st.columns(min(len(images_files), 5))
            for i, f in enumerate(images_files):
                with cols[i % len(cols)]:
                    st.image(Image.open(f), caption=f.name, use_column_width=True)

        if st.button("Register", key="register_btn") and cattle_name and images_files:
            if len(images_files) < 1:
                st.error("Upload at least 1 image.")
            elif not cattle_name.strip():
                st.error("Enter a cattle name.")
            else:
                pil_images = [Image.open(f).convert('RGB') for f in images_files]
                with st.spinner("Registering cattle..."):
                    result = identifier.register_cattle(cattle_name.strip(), pil_images)
                st.success(
                    f"Registered **{result['name']}** with {result['num_images']} images!"
                )

    # --- Tab 4: Registry ---
    with tab4:
        st.header("Registered Cattle")

        cattle_list = identifier.get_registry_list()

        if not cattle_list:
            st.info("No cattle registered yet. Go to 'Register Cattle' tab to add some.")
        else:
            st.write(f"**{len(cattle_list)}** cattle registered:")

            for name in sorted(cattle_list):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{name}**")
                with col2:
                    if st.button("Delete", key=f"del_{name}"):
                        identifier.delete_cattle(name)
                        st.rerun()


if __name__ == "__main__":
    main()
