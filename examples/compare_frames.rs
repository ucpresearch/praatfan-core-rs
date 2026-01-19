use praat_core::Sound;

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();
    let formant = sound.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0);

    println!("Our formant frame values (first 20):");
    println!("frame\ttime\tF1\tF2\tF3");
    for i in 0..20.min(formant.num_frames()) {
        let t = formant.get_time_from_frame(i);
        let f1 = formant.get_value_at_frame(1, i);
        let f2 = formant.get_value_at_frame(2, i);
        let f3 = formant.get_value_at_frame(3, i);
        
        let f1_str = f1.map(|f| format!("{:.1}", f)).unwrap_or("NaN".to_string());
        let f2_str = f2.map(|f| format!("{:.1}", f)).unwrap_or("NaN".to_string());
        let f3_str = f3.map(|f| format!("{:.1}", f)).unwrap_or("NaN".to_string());
        
        println!("{}\t{:.4}\t{}\t{}\t{}", i + 1, t, f1_str, f2_str, f3_str);
    }
}
